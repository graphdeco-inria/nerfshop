/*
    QMVC - Reference Implementation of paper: 
    
    "Mean Value Coordinates for Quad Cages in 3D", 
    jean-Marc Thiery, Pooran Memari and Tamy Boubekeur
    SIGGRAPH Asia 2018
    
    This program allows to compute QMVC for a set of 3D points contained 
    in a cage made of quad and triangles, as well as other flavors of 
    space coordinates for cages (MVC, SMVC, GC, MEC). It comes also with 
    a 3D viewer which helps deforming a mesh with a cage. 
    
    Copyright (C) 2018  jean-Marc Thiery, Pooran Memari and Tamy Boubekeur

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/
#ifndef MVC_H
#define MVC_H





#include <vector>
#include <cmath>
#include <cassert>









namespace MVC3D{


template< class point_t >
inline
double getAngleBetweenUnitVectors( point_t const & u1 , point_t const & u2 ) {
    return 2.0 * asin( (u1 - u2).norm() / 2.0 );
}


// MVC : Code from "Mean Value Coordinates for Closed Triangular Meshes" Schaeffer Siggraph 2005
template< class int_t , class float_t , class point_t >
bool computeCoordinatesOriginalCode(
        point_t const & eta ,
        std::vector< std::vector< int_t > > const & cage_triangles , std::vector< point_t > const & cage_vertices , std::vector< point_t > const & cage_normals ,
        std::vector< float_t > & weights , std::vector< float_t > & w_weights)
{
    typedef typename point_t::type_t    T;
    unsigned int n_vertices = cage_vertices.size() , n_triangles = cage_triangles.size();
    assert( cage_normals.size() == cage_triangles.size()   &&    "cage_normals.size() != cage_triangles.size()" );
    T epsilon = 0.00000001;

    w_weights.clear();
    weights.clear();
    weights.resize( n_vertices , 0.0 );
    T sumWeights = 0.0;

    std::vector< T > d( n_vertices , 0.0 ); std::vector< point_t > u( n_vertices );

    for( unsigned int v = 0 ; v < n_vertices ; ++v ) {
        d[ v ] = ( eta - cage_vertices[ v ] ).norm();
        if( d[ v ] < epsilon ) {
            weights[v] = 1.0;
            return true;
        }
        u[ v ] = ( cage_vertices[v] - eta ) / d[v];
    }

    w_weights.resize(  n_vertices , 0.0 );

    unsigned int vid[3];     T l[3]; T theta[3] ; T w[3]; T c[3]; T s[3];
    for( unsigned int t = 0 ; t < n_triangles ; ++t ) { // the Norm is CCW :
        for( unsigned int i = 0 ; i <= 2 ; ++i ) vid[i] =  cage_triangles[t][i];
        for( unsigned int i = 0 ; i <= 2 ; ++i ) l[ i ] = ( u[ vid[ ( i + 1 ) % 3 ] ] - u[ vid[ ( i + 2 ) % 3 ] ] ).norm();
        for( unsigned int i = 0 ; i <= 2 ; ++i ) theta[i] = 2.0 * asin( l[i] / 2.0 );
        T h = ( theta[0] + theta[1] + theta[2] ) / 2.0;
        if( M_PI - h < epsilon ) { // eta is on the triangle t , use 2d barycentric coordinates :
            for( unsigned int i = 0 ; i <= 2 ; ++i ) w[ i ] = sin( theta[ i ] ) * l[ (i+2) % 3 ] * l[ (i+1) % 3 ];

            sumWeights = w[0] + w[1] + w[2];

            w_weights.clear();
            weights[ vid[0] ] = w[0] / sumWeights;
            weights[ vid[1] ] = w[1] / sumWeights;
            weights[ vid[2] ] = w[2] / sumWeights;
            return true;
        }

        for( unsigned int i = 0 ; i <= 2 ; ++i ) c[ i ] = ( 2.0 * sin(h) * sin(h - theta[ i ]) ) / ( sin(theta[ (i+1) % 3 ]) * sin(theta[ (i+2) % 3 ]) ) - 1.0;

        T sign_Basis_u0u1u2 = 1;
        if( point_t::dot( point_t::cross(u[vid[0]] , u[vid[1]]) , u[vid[2]] ) < 0.0 ) sign_Basis_u0u1u2 = -1;
        for( unsigned int i = 0 ; i <= 2 ; ++i ) s[ i ] = sign_Basis_u0u1u2 * sqrt( std::max<double>( 0.0 , 1.0 - c[ i ] * c[ i ] ) );
        if( fabs( s[0] ) < epsilon   ||   fabs( s[1] ) < epsilon   ||   fabs( s[2] ) < epsilon ) continue; // eta is on the same plane, outside t  ->  ignore triangle t :  
        for( unsigned int i = 0 ; i <= 2 ; ++i ) w[ i ] = ( theta[ i ] - c[ (i+1)% 3 ]*theta[ (i+2) % 3 ] - c[ (i+2) % 3 ]*theta[ (i+1) % 3 ] ) / ( 2.0 * d[ vid[i] ] * sin( theta[ (i+1) % 3 ] ) * s[ (i+2) % 3 ] );

        sumWeights += ( w[0] + w[1] + w[2] );
        w_weights[ vid[0] ] += w[0];
        w_weights[ vid[1] ] += w[1];
        w_weights[ vid[2] ] += w[2];
    }

    for( unsigned int v = 0 ; v < n_vertices ; ++v ) weights[v]  = w_weights[v] / sumWeights;

    return false;
}

// MVC : Code from "Mean Value Coordinates for Closed Triangular Meshes" Schaeffer Siggraph 2005
template< class int_t , class float_t , class point_t >
bool computeCoordinatesCustomCode(
        point_t const & eta ,
        std::vector< int_t > const & cage_triangles , std::vector< point_t > const & cage_vertices , std::vector< point_t > const & cage_normals ,
        std::vector< float_t > & weights , std::vector< float_t > & w_weights)
{
    typedef float_t T;
    unsigned int n_vertices = cage_vertices.size() , n_triangles = cage_triangles.size() / 3;
    T epsilon = 0.00000001;

    w_weights.clear();
    weights.clear();
    weights.resize( n_vertices , 0.0 );
    T sumWeights = 0.0;

    std::vector< T > d( n_vertices , 0.0 ); std::vector< point_t > u( n_vertices );

    for( unsigned int v = 0 ; v < n_vertices ; ++v ) {
        d[ v ] = ( eta - cage_vertices[ v ] ).norm();
        if( d[ v ] < epsilon ) {
            weights[v] = 1.0;
            return true;
        }
        u[ v ] = ( cage_vertices[v] - eta ) / d[v];
    }

    w_weights.resize(  n_vertices , 0.0 );

    unsigned int vid[3];     T l[3]; T theta[3] ; T w[3]; T c[3]; T s[3];
    for( unsigned int t = 0 ; t < n_triangles ; ++t ) { // the Norm is CCW :
        for( unsigned int i = 0 ; i <= 2 ; ++i ) vid[i] =  cage_triangles[3*t+i];
        for( unsigned int i = 0 ; i <= 2 ; ++i ) l[ i ] = ( u[ vid[ ( i + 1 ) % 3 ] ] - u[ vid[ ( i + 2 ) % 3 ] ] ).norm();
        for( unsigned int i = 0 ; i <= 2 ; ++i ) theta[i] = 2.0 * asin( l[i] / 2.0 );
        T h = ( theta[0] + theta[1] + theta[2] ) / 2.0;
        if( M_PI - h < epsilon ) { // eta is on the triangle t , use 2d barycentric coordinates :
            for( unsigned int i = 0 ; i <= 2 ; ++i ) w[ i ] = sin( theta[ i ] ) * l[ (i+2) % 3 ] * l[ (i+1) % 3 ];

            sumWeights = w[0] + w[1] + w[2];

            w_weights.clear();
            weights[ vid[0] ] = w[0] / sumWeights;
            weights[ vid[1] ] = w[1] / sumWeights;
            weights[ vid[2] ] = w[2] / sumWeights;
            return true;
        }

        for( unsigned int i = 0 ; i <= 2 ; ++i ) c[ i ] = ( 2.0 * sin(h) * sin(h - theta[ i ]) ) / ( sin(theta[ (i+1) % 3 ]) * sin(theta[ (i+2) % 3 ]) ) - 1.0;

        T sign_Basis_u0u1u2 = 1;
        if(u[vid[0]].cross(u[vid[1]]).dot(u[vid[2]]) < 0.0 ) sign_Basis_u0u1u2 = -1;
        for( unsigned int i = 0 ; i <= 2 ; ++i ) s[ i ] = sign_Basis_u0u1u2 * sqrt( std::max<double>( 0.0 , 1.0 - c[ i ] * c[ i ] ) );
        if( fabs( s[0] ) < epsilon   ||   fabs( s[1] ) < epsilon   ||   fabs( s[2] ) < epsilon ) continue; // eta is on the same plane, outside t  ->  ignore triangle t :  
        for( unsigned int i = 0 ; i <= 2 ; ++i ) w[ i ] = ( theta[ i ] - c[ (i+1)% 3 ]*theta[ (i+2) % 3 ] - c[ (i+2) % 3 ]*theta[ (i+1) % 3 ] ) / ( 2.0 * d[ vid[i] ] * sin( theta[ (i+1) % 3 ] ) * s[ (i+2) % 3 ] );

        sumWeights += ( w[0] + w[1] + w[2] );
        w_weights[ vid[0] ] += w[0];
        w_weights[ vid[1] ] += w[1];
        w_weights[ vid[2] ] += w[2];
    }

    for( unsigned int v = 0 ; v < n_vertices ; ++v ) weights[v]  = w_weights[v] / sumWeights;

    return false;
}



template< class int_t , class float_t , class point_t >
bool computeCoordinatesOriginalCode(
        point_t const & eta ,
        std::vector< std::vector< int_t > > const & cage_triangles ,
        std::vector< point_t > const & cage_vertices ,
        std::vector< point_t > const & cage_normals ,
        std::vector< float_t > & weights)
{
    std::vector< float_t > w_weights;
    return computeCoordinatesOriginalCode(eta , cage_triangles , cage_vertices , cage_normals , weights , w_weights );
}









template< class int_t , class float_t , class point_t >
bool computeCoordinatesSimpleCode(
        point_t const & eta ,
        std::vector< std::vector< int_t > > const & cage_triangles ,
        std::vector< point_t > const & cage_vertices ,
        std::vector< point_t > const & cage_normals ,
        std::vector< float_t > & weights ,
        std::vector< float_t > & w_weights)
{
    typedef typename point_t::type_t    T;

    T epsilon = 0.000000001;

    unsigned int n_vertices = cage_vertices.size();
    unsigned int n_triangles = cage_triangles.size();

    assert( cage_normals.size() == cage_triangles.size()   &&    "cage_normals.size() != cage_triangles.size()" );
    w_weights.clear();
    weights.clear();
    weights.resize( n_vertices , 0.0 );
    T sumWeights = 0.0;

    std::vector< T > d( n_vertices , 0.0 );
    std::vector< point_t > u( n_vertices );

    for( unsigned int v = 0 ; v < n_vertices ; ++v )
    {
        d[ v ] = ( eta - cage_vertices[ v ] ).norm();
        if( d[ v ] < epsilon )
        {
            weights[v] = 1.0;
            return true;
        }
        u[ v ] = ( cage_vertices[v] - eta ) / d[v];
    }

    w_weights.resize(  n_vertices , 0.0 );

    unsigned int vid[3];
    T l[3]; T theta[3] ; T w[3];

    for( unsigned int t = 0 ; t < n_triangles ; ++t )
    {
        // the Norm is CCW :
        for( unsigned int i = 0 ; i <= 2 ; ++i )
            vid[i] =  cage_triangles[t][i];

        for( unsigned int i = 0 ; i <= 2 ; ++i ) {
            l[ i ] = ( u[ vid[ ( i + 1 ) % 3 ] ] - u[ vid[ ( i + 2 ) % 3 ] ] ).norm();
        }

        for( unsigned int i = 0 ; i <= 2 ; ++i )
        {
         //   theta[i] = 2.0 * asin( l[i] / 2.0 );
            theta[i] = getAngleBetweenUnitVectors( u[ vid[ ( i + 1 ) % 3 ] ] , u[ vid[ ( i + 2 ) % 3 ] ] );
        }

        // test in original MVC paper: (they test if one angle psi is close to 0: it is "distance sensitive" in the sense that it does not
        // relate directly to the distance to the support plane of the triangle, and the more far away you go from the triangle, the worse it is)
        // In our experiments, it is actually not the good way to do it, as it increases significantly the errors we get in the computation of weights and derivatives,
        // especially when evaluating Hfx, Hfy, Hfz which can be of norm of the order of 10^3 instead of 0 (when specifying identity on the cage, see paper)

        // simple test we suggest:
        // the determinant of the basis is 2*area(T)*d( eta , support(T) ), we can directly test for the distance to support plane of the triangle to be minimum
        T determinant = point_t::dot( cage_vertices[vid[0]] - eta , point_t::cross( cage_vertices[vid[1]] - cage_vertices[vid[0]] , cage_vertices[vid[2]] - cage_vertices[vid[0]] ) );
        T sqrdist = determinant*determinant / (4 * point_t::cross( cage_vertices[vid[1]] - cage_vertices[vid[0]] , cage_vertices[vid[2]] - cage_vertices[vid[0]] ).sqrnorm() );
        T dist = sqrt( (T)sqrdist );

        if( dist < epsilon )
        {
            // then the point eta lies on the support plane of the triangle
            T h = ( theta[0] + theta[1] + theta[2] ) / 2.0;
            if( M_PI - h < epsilon )
            {
                // eta lies inside the triangle t , use 2d barycentric coordinates :
                for( unsigned int i = 0 ; i <= 2 ; ++i )
                {
                    w[ i ] = sin( theta[ i ] ) * l[ (i+2) % 3 ] * l[ (i+1) % 3 ];
                }
                sumWeights = w[0] + w[1] + w[2];

                w_weights.clear();
                weights[ vid[0] ] = w[0] / sumWeights;
                weights[ vid[1] ] = w[1] / sumWeights;
                weights[ vid[2] ] = w[2] / sumWeights;
                return true;
            }
        }

        point_t pt[3] , N[3];

        for( unsigned int i = 0 ; i < 3 ; ++i )
            pt[i] = cage_vertices[ cage_triangles[t][i] ];
        for( unsigned int i = 0 ; i < 3 ; ++i )
            N[i] = point_t::cross( pt[(i+1)%3] - eta , pt[(i+2)%3] - eta );

        for( unsigned int i = 0 ; i <= 2 ; ++i )
        {
            w[i] = 0.0;
            for( unsigned int j = 0 ; j <= 2 ; ++j )
                w[i] += theta[j] * point_t::dot( N[i] , N[j] ) / ( 2.0 * N[j].norm() );

            w[i] /= determinant;
        }

        sumWeights += ( w[0] + w[1] + w[2] );
        w_weights[ vid[0] ] += w[0];
        w_weights[ vid[1] ] += w[1];
        w_weights[ vid[2] ] += w[2];
    }

    for( unsigned int v = 0 ; v < n_vertices ; ++v )
        weights[v]  = w_weights[v] / sumWeights;

    return false;
}

template< class int_t , class float_t , class point_t >
bool computeCoordinatesSimpleCode(
        point_t const & eta ,
        std::vector< std::vector< int_t > > const & cage_triangles ,
        std::vector< point_t > const & cage_vertices ,
        std::vector< point_t > const & cage_normals ,
        std::vector< float_t > & weights)
{
    std::vector< float_t > w_weights;
    return computeCoordinatesSimpleCode(eta , cage_triangles , cage_vertices , cage_normals , weights , w_weights );
}









template< class int_t , class float_t , class point_t >
bool computeCoordinates(
        point_t const & eta ,
        std::vector< std::vector< int_t > > const & cage_triangles ,
        std::vector< point_t > const & cage_vertices ,
        std::vector< point_t > const & cage_normals ,
        std::vector< float_t > & weights ,
        std::vector< float_t > & w_weights)// unnormalized weights
{
    // return computeCoordinatesOriginalCode(eta,cage_triangles,cage_vertices,cage_normals,weights,w_weights);
    return computeCoordinatesSimpleCode(eta,cage_triangles,cage_vertices,cage_normals,weights,w_weights);
}



template< class int_t , class float_t , class point_t >
bool computeCoordinates(
        point_t const & eta ,
        std::vector< std::vector< int_t > > const & cage_triangles ,
        std::vector< point_t > const & cage_vertices ,
        std::vector< point_t > const & cage_normals ,
        std::vector< float_t > & weights)
{
    std::vector< float_t > w_weights;
    return computeCoordinates(eta,cage_triangles,cage_vertices,cage_normals,weights,w_weights);
}





} // namespace MVC3D









#endif // MVC_H
